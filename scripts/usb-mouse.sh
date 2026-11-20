#!/usr/bin/env bash
# Add a standard USB HID mouse interface to the existing AidLux gadget.
# No mouse reports are sent. ADB/diag links are left in place.
set -Eeuo pipefail

base=/sys/kernel/config/usb_gadget/g1
config=$base/configs/c.1
hid=$base/functions/hid.codex_mouse
link=$config/codex_mouse
controller=a600000.dwc3
action=${1:-status}

status() {
    printf 'USB state: '; cat "/sys/class/udc/$controller/state"
    if [[ -L $link ]]; then
        printf 'Mouse interface: enabled\n'
        ls -l /dev/hidg* 2>/dev/null || true
    else
        printf 'Mouse interface: not enabled\n'
    fi
}

case "$action" in
    status) status; exit ;;
    enable|disable) ;;
    *) echo "Usage: $0 {status|enable|disable}" >&2; exit 2 ;;
esac
if (( EUID != 0 )); then
    echo 'Run this command with sudo on AidLux.' >&2; exit 1
fi
[[ -d $config && -d /sys/class/udc/$controller ]] || {
    echo 'Expected AidLux gadget or controller not found. Nothing changed.' >&2; exit 1;
}
exec 9>/run/lock/aidlux-usb-mouse.lock
flock -n 9 || { echo 'Another mouse configuration is running.' >&2; exit 1; }
[[ $(<"$base/UDC") == "$controller" ]] || {
    echo 'The existing gadget is not bound to the expected controller. Nothing changed.' >&2; exit 1;
}

# The vendor USB stack uses this marker to suppress automatic rebinding.
marker=/tmp/usb_bind_in_progress
(set -o noclobber; printf 'start\n' > "$marker") || {
    echo 'Vendor USB reconfiguration already in progress. Nothing changed.' >&2; exit 1;
}
trap 'rm -f -- "$marker"' EXIT

if [[ $action == enable ]]; then
    if [[ -L $link && $(readlink -f -- "$link") == "$hid" ]]; then status; exit; fi
    [[ ! -e $hid && ! -e $link && ! -L $link ]] || {
        echo 'Mouse function or link already exists in an unexpected state. Nothing changed.' >&2; exit 1;
    }
    created=0 linked=0 unbound=0
    rollback() {
        rc=$?
        trap - EXIT
        rm -f -- "$marker"
        if (( rc != 0 )); then
            set +e
            if (( linked )); then rm -- "$link"; fi
            if (( unbound )) && [[ $(<"$base/UDC") != "$controller" ]]; then
                printf '%s\n' "$controller" > "$base/UDC" || echo 'Could not restore USB binding.' >&2
            fi
            if (( created )); then rmdir -- "$hid"; fi
        fi
        exit "$rc"
    }
    trap rollback EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mkdir -- "$hid"; created=1
    printf '1\n' > "$hid/subclass"
    printf '2\n' > "$hid/protocol"
    printf '3\n' > "$hid/report_length"
    # Mouse: three buttons, five padding bits, then relative X/Y int8.
    printf '%b' '\x05\x01\x09\x02\xa1\x01\x09\x01\xa1\x00\x05\x09\x19\x01\x29\x03\x15\x00\x25\x01\x95\x03\x75\x01\x81\x02\x95\x01\x75\x05\x81\x03\x05\x01\x09\x30\x09\x31\x15\x81\x25\x7f\x75\x08\x95\x02\x81\x06\xc0\xc0' > "$hid/report_desc"
    printf 'none\n' > "$base/UDC"; unbound=1
    binding=$(<"$base/UDC")
    [[ -z $binding || $binding == none ]] || { printf 'USB controller still bound: %s\n' "$binding" >&2; exit 1; }
    (cd "$base" && ln -s -- "functions/${hid##*/}" "configs/c.1/${link##*/}"); linked=1
    printf '%s\n' "$controller" > "$base/UDC"; unbound=0
    rm -f -- "$marker"
    trap - EXIT INT TERM
else
    if [[ ! -L $link && ! -e $hid ]]; then status; exit; fi
    [[ -L $link && $(readlink -f -- "$link") == "$hid" ]] || {
        echo 'Mouse link is in an unexpected state. Nothing changed.' >&2; exit 1;
    }
    removed=0 unbound=0
    rollback() {
        rc=$?
        trap - EXIT
        rm -f -- "$marker"
        if (( rc != 0 && unbound )); then
            set +e
            if (( removed )); then (cd "$base" && ln -s -- "functions/${hid##*/}" "configs/c.1/${link##*/}"); fi
            printf '%s\n' "$controller" > "$base/UDC" || echo 'Could not restore USB binding.' >&2
        fi
        exit "$rc"
    }
    trap rollback EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf 'none\n' > "$base/UDC"; unbound=1
    rm -- "$link"; removed=1
    printf '%s\n' "$controller" > "$base/UDC"; unbound=0
    rm -f -- "$marker"
    trap - EXIT INT TERM
    rmdir -- "$hid" || echo 'Interface removed; unused function directory remains.' >&2
fi
for (( i=0; i<20; i++ )); do
    [[ $(<"/sys/class/udc/$controller/state") == configured ]] && break
    sleep 0.2
done
status
