undef $/; $_ = <STDIN>;
my $new = <<'NEW';
    keep = sorted(set(list(range(0, int(dn), 12)) + [int(dn) - 1]))   # D145: the kept frames only, no full archive
    np.savez(f"{args.out}_frames12.npz", frames=np.stack([frames.x[k] for k in keep]), raws=np.asarray(keep),
             tgt=tgt, deliver_n=np.int64(dn),
             **({} if app is None else dict(src_albedo=app.src, tgt_albedo=app.tgt,
                                            appearance=np.str_(json.dumps(app.params)))))
NEW
s/    np\.savez\(f"\{args\.out\}_\{ARM\}\.npz".*?appearance=np\.str_\(json\.dumps\(app\.params\)\)\)\)\)\n/$new/s or die "no match";
print;
